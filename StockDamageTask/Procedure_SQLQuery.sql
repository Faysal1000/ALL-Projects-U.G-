CREATE OR ALTER PROCEDURE dbo.SP_StockDamage_Save
    @VoucherNo INT = NULL,         
    @GodownNo NVARCHAR(50) = NULL,
    @ItemName NVARCHAR(100) = NULL,
    @ItemCode NVARCHAR(50) = NULL,
    @BatchNo INT = NULL,
    @CurrencyName NVARCHAR(50) = NULL,
    @ConversionRate DECIMAL(18,6) = NULL,
    @StockQuantity DECIMAL(10,3) = NULL,
    @Quantity DECIMAL(10,0) = NULL,
    @Rate DECIMAL(10,3) = NULL,
    @Amount_in DECIMAL(10,3) = NULL,
    @DrACHead NVARCHAR(100) = 'Stock Damage',
    @EmployeeId INT = NULL,
    @CreatedDate DATETIME = NULL,
    @Comments NVARCHAR(500) = NULL,
    @TotalAmount DECIMAL(10,3) = NULL,
    @DeleteVoucher BIT = 0
AS
BEGIN
    SET NOCOUNT ON;

    IF @CreatedDate IS NULL
        SET @CreatedDate = GETDATE();

    BEGIN TRANSACTION;
    BEGIN TRY
        -- DELETE 
        IF @DeleteVoucher = 1
        BEGIN
            IF EXISTS (SELECT 1 FROM dbo.StockDamage WHERE VoucherNo = @VoucherNo)
            BEGIN
                DELETE FROM dbo.StockDamage WHERE VoucherNo = @VoucherNo;

                COMMIT TRANSACTION;
                SELECT 'Deleted' AS Status, 'Voucher deleted successfully.' AS Message;
            END
            ELSE
            BEGIN
                ROLLBACK TRANSACTION;
                SELECT 'Error' AS Status, 'Voucher number not found for deletion.' AS Message;
            END
            RETURN;
        END

        -- UPDATE 
        IF @VoucherNo IS NOT NULL AND EXISTS (SELECT 1 FROM dbo.StockDamage WHERE VoucherNo = @VoucherNo)
        BEGIN
            UPDATE dbo.StockDamage
            SET 
                GodownNo = @GodownNo,
                ItemName = @ItemName,
                ItemCode = @ItemCode,
                BatchNo = @BatchNo,
                CurrencyName = @CurrencyName,
                ConversionRate = @ConversionRate,
                StockQuantity = @StockQuantity,
                Quantity = @Quantity,
                Rate = @Rate,
                Amount_in = @Amount_in,
                DrACHead = @DrACHead,
                EmployeeId = @EmployeeId,
                CreatedDate = @CreatedDate,
                Comments = @Comments,
                TotalAmount = @TotalAmount
            WHERE VoucherNo = @VoucherNo;

            UPDATE s
            SET s.Quantity = s.Quantity - @Quantity
            FROM dbo.Stock s
            WHERE s.SubItemCode = @ItemCode;

            COMMIT TRANSACTION;
            SELECT 'Updated' AS Status, 'Stock damage updated successfully.' AS Message;
            RETURN;
        END

        -- INSERT 
        INSERT INTO dbo.StockDamage
        (
            GodownNo, ItemName, ItemCode, BatchNo,
            CurrencyName, ConversionRate, StockQuantity,
            Quantity, Rate, Amount_in, DrACHead, EmployeeId,
            CreatedDate, Comments, TotalAmount
        )
        VALUES
        (
            @GodownNo, @ItemName, @ItemCode, @BatchNo,
            @CurrencyName, @ConversionRate, @StockQuantity,
            @Quantity, @Rate, @Amount_in, @DrACHead, @EmployeeId,
            @CreatedDate, @Comments, @TotalAmount
        );

        DECLARE @NewVoucherNo INT = SCOPE_IDENTITY();

        UPDATE s
        SET s.Quantity = s.Quantity - @Quantity
        FROM dbo.Stock s
        WHERE s.SubItemCode = @ItemCode;

        COMMIT TRANSACTION;
        SELECT 'Inserted' AS Status, CONCAT('Stock damage recorded successfully. VoucherNo = ', @NewVoucherNo) AS Message, @NewVoucherNo AS VoucherNo;
    END TRY

    BEGIN CATCH
        ROLLBACK TRANSACTION;
        SELECT 'Error' AS Status, ERROR_MESSAGE() AS Message;
    END CATCH
END

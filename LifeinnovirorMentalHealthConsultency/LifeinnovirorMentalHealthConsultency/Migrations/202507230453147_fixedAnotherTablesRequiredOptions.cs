namespace LifeinnovirorMentalHealthConsultency.Migrations
{
    using System;
    using System.Data.Entity.Migrations;
    
    public partial class fixedAnotherTablesRequiredOptions : DbMigration
    {
        public override void Up()
        {
            AlterColumn("dbo.Doctors", "Status", c => c.String(maxLength: 20));
            AlterColumn("dbo.DoctorStatusHistories", "Status", c => c.String(maxLength: 20));
        }
        
        public override void Down()
        {
            AlterColumn("dbo.DoctorStatusHistories", "Status", c => c.String(nullable: false, maxLength: 20));
            AlterColumn("dbo.Doctors", "Status", c => c.String(nullable: false, maxLength: 20));
        }
    }
}

namespace LifeinnovirorMentalHealthConsultency.Migrations
{
    using System;
    using System.Data.Entity.Migrations;
    
    public partial class fixedTablesRequiredOptions : DbMigration
    {
        public override void Up()
        {
            AlterColumn("dbo.LeaveRequests", "Status", c => c.String(maxLength: 20));
            AlterColumn("dbo.Payments", "Status", c => c.String(maxLength: 20));
            AlterColumn("dbo.RegistrationRequests", "Status", c => c.String(maxLength: 20));
        }
        
        public override void Down()
        {
            AlterColumn("dbo.RegistrationRequests", "Status", c => c.String(nullable: false, maxLength: 20));
            AlterColumn("dbo.Payments", "Status", c => c.String(nullable: false, maxLength: 20));
            AlterColumn("dbo.LeaveRequests", "Status", c => c.String(nullable: false, maxLength: 20));
        }
    }
}
